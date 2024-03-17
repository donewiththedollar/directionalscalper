import uuid
from .exchange import Exchange  # Assuming exchange.py is in the same directory
import logging

class BybitExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('bybit', api_key, secret_key, passphrase, market_type)
    
    def transfer_funds(self, code: str, amount: float, from_account: str, to_account: str, params={}):
        """
        Transfer funds between different account types under the same UID.
        """
        try:
            # Generate a unique transfer ID (UUID)
            transfer_id = str(uuid.uuid4())

            # Add the transfer ID to the params dictionary
            params['transferId'] = transfer_id

            # Use CCXT's transfer function to initiate the internal transfer
            transfer = self.exchange.transfer(code, amount, from_account, to_account, params)

            if transfer:
                logging.info(f"Funds transfer successful. Details: {transfer}")
                return transfer
            else:
                logging.error("Error occurred during funds transfer.")
                return None
        except Exception as e:
            logging.error(f"Error occurred during funds transfer: {e}")
            return None

    def get_bybit_wallet_balance(self, coin: str):
        """
        Fetch the Bybit wallet balance for a specific coin.
        """
        try:
            # Use the fetch_balance method from CCXT (inherited from the Exchange class)
            balance = self.exchange.fetch_balance()

            if coin in balance:
                return balance[coin]['free']
            else:
                logging.warning(f"Coin {coin} not found in Bybit wallet balance.")
                return None
        except Exception as e:
            logging.error(f"Error occurred while fetching Bybit wallet balance: {e}")
            return None

    def get_futures_balance_bybit(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type if needed
                balance_response = self.exchange.fetch_balance({'type': 'swap'})

                # Log the raw response for debugging purposes
                #logging.info(f"Raw balance response from Bybit: {balance_response}")

                # Parse the balance
                if quote in balance_response['total']:
                    total_balance = balance_response['total'][quote]
                    return total_balance
                else:
                    logging.warning(f"Balance for {quote} not found in the response.")
            except Exception as e:
                logging.error(f"Error fetching balance from Bybit: {e}")

        return None

    def get_available_balance_bybit(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type
                balance_response = self.exchange.fetch_balance({'type': 'swap'})

                # Log the raw response for debugging purposes
                #logging.info(f"Raw available balance response from Bybit: {balance_response}")

                # Check for the required keys in the response
                if 'free' in balance_response and quote in balance_response['free']:
                    # Return the available balance for the specified currency
                    return float(balance_response['free'][quote])
                else:
                    logging.warning(f"Available balance for {quote} not found in the response.")

            except Exception as e:
                logging.error(f"Error fetching available balance from Bybit: {e}")

        return None
    